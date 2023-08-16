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
from collections import defaultdict
from random import sample
import pandas as pd
import pickle


class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset. Tailored for the Open
    World Benchmark.
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
        print(self._image_set_path)
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
        Evaluation code at inference for known precision and unknown recall
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """

        unknown_class_index = 80
        ONLY_PREDICT = True
        PREVIOUS_KNOWN = 0
        NUM_CLASSES = PREVIOUS_KNOWN + 80
        UNKNOWN = True
        SAVE_SCORES = False
        # For f1 pseudo calculation
        SAVE_ALL_SCORES = False

        UPPER_THRESH = 100
        PSEUDO_LABEL_KNOWN = False
        if PSEUDO_LABEL_KNOWN:
            UPPER_THRESH = 55
        SINGLE_BRANCH = True
        known_removal = False
        predict_fn = "predictions/t1/known_dual_test.pickle"
        #predict_fn = "predictions/t2/known_dual_test.pickle"
        tpfp_fn = "t2_known_tpfp_scores.csv"

        all_predictions = comm.gather(self._predictions, dst=0)
        # list containing dictionary of keys with classes and values predictions
        # each prediction contains [image id, score, xmin, ymin, xmax, ymax
        import pandas as pd

        if ONLY_PREDICT:
            ids = [];probs = [];xmin = [];ymin = [];xmax = [];ymax = []
            classes = []
            for predictions_per_gpu in all_predictions:
                for clsid, lines in predictions_per_gpu.items():
                    for p in lines:
                        ps = p.split(" ")
                        ids.append(ps[0])
                        probs.append(ps[1])
                        xmin.append(ps[2])
                        ymin.append(ps[3])
                        xmax.append(ps[4])
                        ymax.append(ps[5])    
                        classes.append(80)
            df = pd.DataFrame();df["ids"] = ids;df["probs"] = probs;df["xmin"] = xmin;df["ymin"] = ymin;df["xmax"] = xmax; df["ymax"] = ymax
            df["class"] = classes
            df["ids"] = df["ids"].astype('str')
            df.to_csv("unknown_t1_predictions.csv")
            return
        # if ONLY_PREDICT:
        #     image_prediction_hash = defaultdict(list)
        #     for predictions_per_gpu in all_predictions:
        #         for clsid, lines in predictions_per_gpu.items():
        #             for line in lines:
        #                 line_split = line.split(" ")
        #                 image_id = line_split[0]
        #                 image_prediction_hash[image_id].append([clsid] + line_split[1:])
        #     with open(predict_fn, 'wb') as handle:
        #         print("saved predictions", predict_fn)
        #         pickle.dump(image_prediction_hash, handle)
        #     return
        # key: imageid, value: predictions
        # predictions: class, score, xmin, ymin, xmax, ymax
        general_predictions_hash = defaultdict(list)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                if SINGLE_BRANCH:
                    # Always predict unknown for single branch
                    predictions[0].extend(lines)
                else:
                    # Put all the predictions above the number of classes to the 80th class
                    if clsid < NUM_CLASSES:
                        predictions[clsid].extend(lines)
                    else:
                        predictions[unknown_class_index].extend(lines)
        del all_predictions
        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")
            aps = defaultdict(list)  # iou -> ap per class
            recs = {}
            recs[50] = []

            if UNKNOWN:
                for cls_id, cls_name in enumerate(self._class_names[1:2]):
                    cls_id = 0
                    cls_name = "aeroplane"
                    lines = predictions.get(cls_id, [""])
                    with open(res_file_template.format(cls_name), "w") as f:
                        f.write("\n".join(lines))
                    for thresh in range(50, 55, 5):
                        rec, prec, ap = owod_eval(
                            res_file_template,
                            self._anno_file_template,
                            self._image_set_path,
                            cls_name,
                            ovthresh=thresh / 100.0,
                            use_07_metric=self._is_2007, known_removal=known_removal, known_pred_fn=predict_fn
                        )
                        #aps[thresh].append(ap * 100)
                        aps[thresh].append(ap)
                        print("recall", rec[-1])
                        if thresh == 50 and cls_id == 0:
                            recs[50] = rec[-1]
                ret = OrderedDict()
                ret["bbox"] = {"unknown_recall50": recs[50]}
                return ret
            else:
                ret = OrderedDict()
                unknown_recall_50 = 0
                start_index = 0
                # Use previous class f scores
                # randomly sample 1000 for tpfp generation
                #if SAVE_ALL_SCORES:
                #    start_index = PREVIOUS_KNOWN
                for cls_id, cls_name in enumerate(self._class_names[start_index:NUM_CLASSES]):
                    lines = predictions.get(cls_id, [""])
                    with open(res_file_template.format(cls_name), "w") as f:
                        f.write("\n".join(lines))
                    if SAVE_SCORES:
                        save_class_scores(predictions)
                        return 1
                    for thresh in range(50, UPPER_THRESH, 5):
                        # thresholds 50, 55, ...95
                        # For debugging purposes
                        if len(lines) == 1:
                            rec, ap = [0], 0
                        else:
                            rec, prec, ap = voc_eval(
                                res_file_template,
                                self._anno_file_template,
                                self._image_set_path,
                                cls_name,
                                ovthresh=thresh / 100.0,
                                use_07_metric=self._is_2007, df_classes=self.df_classes,
                            df_probs=self.df_probs,
                            df_tp=self.df_tp,
                            df_save=SAVE_ALL_SCORES, unknown=False, pseudo_knowns=False
                            )
                        if cls_id != unknown_class_index:
                            aps[thresh].append(ap * 100)
                        if thresh == 50 and cls_id == unknown_class_index:
                            unknown_recall_50 = rec[-1]
                    if cls_id == PREVIOUS_KNOWN - 1:
                        map_prev = {iou: np.mean(x) for iou, x in aps.items()}
                        ret["bbox_prev"] = {"AP": np.mean(list(map_prev.values())),
                                            "AP50": map_prev[50]}
                    if SAVE_ALL_SCORES:
                        self.df["classes"] = self.df_classes
                        self.df["probs"] = self.df_probs
                        self.df["tp"] = self.df_tp
                        self.df.to_csv(tpfp_fn)
                        print("saved tpfp scores")
                        return
                map_current = np.mean(aps[50][PREVIOUS_KNOWN:NUM_CLASSES])
                ret["bbox_current"] = {"AP50": map_current}
                mAP = {iou: np.mean(x) for iou, x in aps.items()}
                ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
                ret["recall"] = {"UnknownRecall50": unknown_recall_50}
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
    """Parse a PASCAL VOC xml file.
    Used for loading the annotations for test evaluation
    Unlike at train time, we load all of the labels"""
    VOC_CLASS_NAMES_COCOFIED = [
        "airplane", "dining table", "motorcycle",
        "potted plant", "couch", "tv"
    ]
    BASE_VOC_CLASS_NAMES = [
        "aeroplane", "diningtable", "motorbike",
        "pottedplant", "sofa", "tvmonitor"
    ]
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        cls_name = obj.find("name").text
        obj_struct["name"] = cls_name
        if cls_name in VOC_CLASS_NAMES_COCOFIED:
            obj_struct["name"] = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls_name)]
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


"""
    Non-max Suppression Algorithm
    @param list  Object candidate bounding boxes
    @param list  Confidence score of bounding boxes
    @param float IoU threshold
    @return Rest boxes after nms operation
"""


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


def iou(BBGT, bb):
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
    return ovmax, jmax


def owod_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, gph=None, known_removal=False, known_pred_fn="predictions/t1/known_dual_test.pickle"):
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

    OWDETR_T1_CLASS_NAMES = {
        "aeroplane","bicycle","bird","boat","bus","car",
        "cat","cow","dog","horse","motorbike","sheep","train",
        "elephant","bear","zebra","giraffe","truck","person"
    }

    OWDETR_T2_CLASS_NAMES = {
        "traffic light","fire hydrant","stop sign",
        "parking meter","bench","chair","diningtable",
        "pottedplant","backpack","umbrella","handbag",
        "tie","suitcase","microwave","oven","toaster","sink",
        "refrigerator","bed","toilet","sofa"
    }

    OWDETR_T3_CLASS_NAMES = {
        "frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard",
        "surfboard","tennis racket","banana","apple","sandwich",
        "orange","broccoli","carrot","hot dog","pizza","donut","cake"
    }

    OWDETR_T4_CLASS_NAMES = {
        "laptop","mouse","remote","keyboard","cell phone","book",
        "clock","vase","scissors","teddy bear","hair drier","toothbrush",
        "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
    }

    #OWDETR_T4_CLASS_NAMES.update(OWDETR_T3_CLASS_NAMES)
    #OWDETR_T4_CLASS_NAMES.update(OWDETR_T2_CLASS_NAMES)

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
    # extract gt objects for this class
    class_recs = {}
    # number of positive instances for the class
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] in T4_CLASS_NAMES]
        # Get known rectangles only for t1
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
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
    # For each bounding box
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # 1 if not overlapping, used as a mask to select predictions that don't overlap with ground truth
    non_overlap_indices = np.array([True] * nd)

    known_hash = defaultdict(list)
    if known_removal:
        with open(known_pred_fn, 'rb') as handle:
            known_saved_hash = pickle.load(handle)
            for key, value in known_saved_hash.items():
                for prediction in value:
                    known_hash[key].append([float(prediction[2]), float(prediction[3]), float(prediction[4]),
                                            float(prediction[5])])
    unknown_hash = defaultdict(list)
    for key, value in zip(image_ids, BB):
        unknown_hash[key].append(value)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        # Can contain multiple bounding boxes in the ground truth
        # BB contains all of the objects
        # iou returns the max iou for each pair
        BBGT = R["bbox"].astype(float)
        known_pred_boxes = None
        if image_ids[d] in known_hash.keys():
            known_pred_boxes = known_hash[image_ids[d]]
            known_pred_boxes = np.array([elem for elem in known_pred_boxes]).astype(np.float)
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
                    flag = False
                    if flag:
                        o, j = iou(known_pred_boxes, bb)
                        if o > 0.9:
                            #print("known overlap")
                            fp[d] = 1.0
                        else:
                            tp[d] = 1.0
                            R["det"][jmax] = 1
                            # don't add the prediction as a potential label
                            non_overlap_indices[d] = False
                            #print("detected unknown")
                            R["det"][jmax] = 1
                    else:
                        tp[d] = 1.0
                        R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    # We just take the last index of the rec matrix
    return rec, prec, ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, df_classes=None,
             df_probs=None, df_tp=None, df_save=False, unknown=False, pseudo_knowns=False):
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
    PSEUDO_KNOWNS = pseudo_knowns

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
    # number of positive instances for the class
    npos = 0
    for imagename in imagenames:
        if classname == "unknown":
            R = [obj for obj in recs[imagename] if obj["name"] in T4_CLASS_NAMES]
        else:
            R = [obj for obj in recs[imagename] if obj["name"] == classname]
        # Get known rectangles only for t1
        # if classname == "aeroplane":
        #     R = [obj for obj in recs[imagename] if obj["name"] not in T4_CLASS_NAMES]
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
    # For each bounding box
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # 1 if not overlapping, used as a mask to select predictions that don't overlap with ground truth
    non_overlap_indices = np.array([True] * nd)

    # For each image id key, value is a list of bounding boxes
    image_id_boxes = defaultdict(list)
    image_id_scores = defaultdict(list)
    # For each image id key, value is a list of bounding boxes
    image_id_boxes = defaultdict(list)
    image_id_scores = defaultdict(list)

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
                    if df_save:
                        df_tp.append(True)
                        df_probs.append(sorted_conf[d])
                        df_classes.append(classname)
                    # don't add the prediction as a potential label
                    non_overlap_indices[d] = False
                    #print("detected unknown")
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
                    if df_save:
                        df_tp.append(False)
                        df_probs.append(sorted_conf[d])
                        df_classes.append(classname)
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
            with open("pseudolabels/t2/known_50_2/boxes_" + str(classname) + ".pickle", 'wb') as handle:
                pickle.dump(image_ids_nms_boxes, handle)
            with open("pseudolabels/t2/known_50_2/scores_" + str(classname) + ".pickle", 'wb') as handle:
                pickle.dump(image_ids_nms_scores, handle)
            print(classname, npos, len(image_ids_nms_boxes))
            # with open("pseudolabels/t2/known/tpscores_" + str(classname) + ".pickle", 'wb') as handle:
            #     pickle.dump(class_scores, handle)
            print("saved pseudo knowns")
        #if tp[d] == 0:
            # Add the bounding box prediction if there was no ground truth overlap

            # image_id_boxes[image_ids[d]].append(bb)
            # image_id_scores[image_ids[d]].append(sorted_conf[d])
    # image_ids_nms_boxes = {}
    # image_ids_nms_scores = {}
    # for key in image_id_boxes.keys():
    #     bounding_boxes = image_id_boxes[key]
    #     #bounding_boxes = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)]
    #     #confidence_score = [0.9, 0.75, 0.8]
    #     confidence_score = np.array(image_id_scores[key])
    #     picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold=ovthresh)
    #     image_ids_nms_boxes[key] = picked_boxes
    #     image_ids_nms_scores[key] = picked_score
    # with open('pseudolabels/boxes.pickle', 'wb') as handle:
    #     pickle.dump(image_ids_nms_boxes, handle)
    # with open('pseudolabels/scores.pickle', 'wb') as handle:
    #     pickle.dump(image_ids_nms_scores, handle)
    # with open('pseudolabels/scores.pickle', 'rb') as handle:
    #     bp = pickle.load(handle)
    #     print(image_ids_nms_scores == bp)
    # return

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    # We just take the last index of the rec matrix
    return rec, prec, ap
