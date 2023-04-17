# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator

import selectivesearch
from PIL import Image


class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)

        # Too many tiny files, download all to local for speed.
        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings


    # Called by detectron2/evaluation.evaluator.py
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions
        #breakpoint()
        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            # For each class, calculate the ap
            #for cls_id, cls_name in enumerate(self._class_names):
                # if cls_id != 80:
                #     continue
            # first 20 for task 1
            recs = {}
            recs[50] = []
            for cls_id, cls_name in enumerate(self._class_names[1:2]):
                cls_id = 0
                cls_name = "aeroplane"
                #print(cls_id, cls_name)
                lines = predictions.get(cls_id, [""])
                #breakpoint()
                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 80, 25):
                #for thresh in range(50, 80, 25):
                    # thresholds 50, 55, ...95
                    #print("evaluating at threshold", thresh)
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)
                    #print("recall", rec[-1])
                    if thresh == 50 and cls_id == 0:
                        recs[50] = rec[-1]
                    return
                #breakpoint()
        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        # returns dictionary of keys(metrics) and values
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75],
        "unknown_recall50":recs[50]}
        # the return values get put into metrics.json
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        # This causes problems, because coco doesn't have pose annotations possibly
        #obj_struct["pose"] = obj.find("pose").text
        #obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    T2_CLASS_NAMES = {
        "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "microwave", "oven", "toaster", "sink", "refrigerator"
    }

    T3_CLASS_NAMES = {
        "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake"
    }

    T4_CLASS_NAMES = {
        "bed", "toilet", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl"
    }

    T4_CLASS_NAMES.update(T3_CLASS_NAMES)
    T4_CLASS_NAMES.update(T2_CLASS_NAMES)
    #breakpoint()
    # first load gt
    # read list of images
    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # Selective search on the imagenames

    import skimage.data
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    image_proposals = {}
    for count, fn in enumerate(imagenames):
        print(count, fn)
        #breakpoint()
        img_fn = "../PROB/data/VOC2007/JPEGImages/{}.jpg".format(fn)
        pil_im = Image.open(img_fn)
        img = np.asarray(pil_im)
        if len(img.size) == 2:
            print("converting grayscale")
            img = img.convert("RGB")
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
        candidates = set()
        # need to change the format from (xmin, ymin, width, height) to
        # (xmin, ymin, xmax, ymax)
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            # if r['size'] < 2000:
            #     continue
            # # distorted rects
            x, y, w, h = r['rect']
            # if w / h > 1.2 or h / w > 1.2:
            #     continue
            candidates.add((x, y, x + w, y + h))
            #candidates.add(r['rect'])
        image_proposals[fn] = candidates
        # draw rectangles on the original image
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(img)
        # for x, y, w, h in candidates:
        #     print(x, y, w, h)
        #     rect = mpatches.Rectangle(
        #         (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        #     ax.add_patch(rect)

        # plt.show()
    # load annots
    recs = {}
    # some annotations missing for the image
    for imagename in imagenames:
        try:
            recs[imagename] = parse_rec(annopath.format(imagename))
        except:
            print("error with file", imagename)
            print(annopath.format(imagename))
            breakpoint()
    # extract gt objects for this class
    class_recs = {}
    # number of positive instances for the class
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        if classname == "aeroplane":
            R = [obj for obj in recs[imagename] if obj["name"] in T4_CLASS_NAMES]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    # if classname == "unknown" and ovthresh == 0.5:
    #     breakpoint()
    # read detetections
    # detfile = detpath.format(classname)
    # with open(detfile, "r") as f:
    #     lines = f.readlines()

    # splitlines = [x.strip().split(" ") for x in lines]
    # image_ids = [x[0] for x in splitlines]
    # confidence = np.array([float(x[1]) for x in splitlines])
    # BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    # sorted_ind = np.argsort(-confidence)
    # BB = BB[sorted_ind, :]
    # image_ids = [image_ids[x] for x in sorted_ind]
    # sorted_conf = [confidence[x] for x in sorted_ind]
    # go down dets and mark TPs and FPs
    #nd = len(image_ids)
    
    nd = 0
    image_ids = []
    bb_selective_search = []
    for k, v in image_proposals.items():
        num_d = len(v)
        nd += num_d
        bb_selective_search.extend(list(v))
        for n in range(num_d):
            image_ids.append(k)
    # For each bounding box
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    #breakpoint()
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = bb_selective_search[d]
        #bb = BB[d, :].astype(float)
        ovmax = -np.inf
        # Can contain multiple bounding boxes in the ground truth
        # BB contains all of the objects
        # iou returns the max iou for each pair
        BBGT = R["bbox"].astype(float)

        # If there is a groundtruth object for this class
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        #print("ovmax", ovmax)
        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    #print("detected unknown")
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    print(rec[-1])
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    # We just take the last index of the rec matrix
    return rec, prec, ap
