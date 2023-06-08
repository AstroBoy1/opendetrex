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
import pickle
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from random import sample
from detectron2.evaluation.evaluator import DatasetEvaluator
import pickle
import pandas as pd


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

        # For calculating F1 threshold for incremental pseudolabels
        self.df = pd.DataFrame()
        self.df_classes = []
        self.df_probs = []
        self.df_tp = []

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

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )
        PREVIOUS_KNOWN = 0
        NUM_CLASSES = 20
        SAVE_SCORES = False
        ret = OrderedDict()
        # For saving probabilities for tp/fp for each class as a dataframe
        SAVE_ALL_SCORES = False
        UPPER_THRESH = 100
        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            # For each class, calculate the ap
            #for cls_id, cls_name in enumerate(self._class_names):
            # first 20 for task 1
            #breakpoint()
            #with open("t1_known_predictions.pickle", "wb") as f:
                #pickle.dump(predictions, f)
            for cls_id, cls_name in enumerate(self._class_names[:NUM_CLASSES]):
                lines = predictions.get(cls_id, [""])
                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))
                if SAVE_SCORES:
                    save_class_scores(predictions)
                    return 1
                #for thresh in range(50, 55, 5):
                for thresh in range(50, UPPER_THRESH, 5):
                    # thresholds 50, 55, ...95
                    #print("evaluating at threshold", thresh)
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007, df_classes=self.df_classes,
                        df_probs=self.df_probs,
                        df_tp=self.df_tp,
                        df_save=SAVE_ALL_SCORES,
                    )
                    aps[thresh].append(ap * 100)
                if cls_id == PREVIOUS_KNOWN - 1:
                    map_prev = {iou: np.mean(x) for iou, x in aps.items()}
                    ret["bbox_prev"] = {"AP": np.mean(list(map_prev.values())),
                                        "AP50": map_prev[50], "AP75": map_prev[75]}
        if SAVE_ALL_SCORES:
            self.df["classes"] = self.df_classes
            self.df["probs"] = self.df_probs
            self.df["tp"] = self.df_tp
            self.df.to_csv("t1_tpfp_scores.csv")
            return ret
        map_current = np.mean(aps[50][PREVIOUS_KNOWN:])
        ret["bbox_current"] = {"AP50": map_current}
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
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


def save_class_scores(predictions):
    breakpoint()
    with open('t1_class_ratios.pickle', 'rb') as handle:
        t1_dictionary = pickle.load(handle)
    all_files = set()
    for k, v in t1_dictionary.items():
        lines = predictions.get(int(k), [""])
        splitlines = [x.strip().split(" ") for x in lines]
        image_ids = [x[0] for x in splitlines]
        fine_files = sample(image_ids, int(v))
        all_files.update(fine_files)
    return all_files


def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


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


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, df_classes=None,
             df_probs=None, df_tp=None, df_save=False):
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

    # first load gt
    # read list of images
    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

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
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read detetections
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    sorted_conf = [confidence[x] for x in sorted_ind]
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    #breakpoint()
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    PSEUDO_KNOWNS = False
    # For each image id key, value is a list of bounding boxes
    image_id_boxes = defaultdict(list)
    image_id_scores = defaultdict(list)
    class_scores = []
    class_thresholds_df = pd.read_csv("t1_known_class_f1_thresholds.csv")
    #breakpoint()
    class_threshold = class_thresholds_df.loc[class_thresholds_df["class"] == classname]["threshold"].values

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
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
        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    #class_scores.append(sorted_conf[d])
                    if df_save:
                        df_tp.append(True)
                        df_probs.append(sorted_conf[d])
                        df_classes.append(classname)
                    R["det"][jmax] = 1
                else:
                    #class_scores.append(sorted_conf[d])
                    if df_save:
                        df_tp.append(False)
                        df_probs.append(sorted_conf[d])
                        df_classes.append(classname)
                    fp[d] = 1.0
        else:
            if df_save:
                df_tp.append(False)
                df_probs.append(sorted_conf[d])
                df_classes.append(classname)
            fp[d] = 1.0
        if PSEUDO_KNOWNS:
            class_threshold = 0.5
            if sorted_conf[d] >= class_threshold:
                image_id_boxes[image_ids[d]].append(bb)
                image_id_scores[image_ids[d]].append(sorted_conf[d])
    if PSEUDO_KNOWNS:
        image_ids_nms_boxes = {}
        image_ids_nms_scores = {}
        for key in image_id_boxes.keys():
            bounding_boxes = image_id_boxes[key]
            confidence_score = np.array(image_id_scores[key])
            picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold=ovthresh)
            image_ids_nms_boxes[key] = picked_boxes
            image_ids_nms_scores[key] = picked_score
        #breakpoint()
        with open("pseudolabels/t2/known_50/boxes_" + str(classname) + ".pickle", 'wb') as handle:
            pickle.dump(image_ids_nms_boxes, handle)
        with open("pseudolabels/t2/known_50/scores_" + str(classname) + ".pickle", 'wb') as handle:
            pickle.dump(image_ids_nms_scores, handle)
        print(classname, npos, len(image_ids_nms_boxes))
        # with open("pseudolabels/t2/known/tpscores_" + str(classname) + ".pickle", 'wb') as handle:
        #     pickle.dump(class_scores, handle)
        print("saved pseudo knowns")
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap
