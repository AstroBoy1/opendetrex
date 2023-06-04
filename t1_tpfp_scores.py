import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from collections import defaultdict
from detectron2.utils.file_io import PathManager
import itertools
import seaborn as sns


CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

UNK_CLASS = ["unknown"]

VOC_CLASS_NAMES = [
"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]


VOC_COCO_CLASS_NAMES = {}
# Used for the original dataset benchmark
VOC_COCO_CLASS_NAMES["TOWOD"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))


def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)
    num_files = len(fileids)
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    class_counts = defaultdict(int)
    class_files = defaultdict(set)
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
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            class_counts[str(class_names.index(cls))] += 1
            class_files[str(class_names.index(cls))].add(fileid)
    return class_counts, num_files, class_files


def fscore(thresh, df_view, num_positive=10000):
    total = df_view.loc[df_view["probs"] >= thresh]
    precision = len(total.loc[total["tp"] == True]) / len(total)
    recall = len(total.loc[total["tp"] == True]) / num_positive
    #print(precision, recall)
    return precision, recall, (precision * recall) / (precision + recall)


def main():
    thresholds = [x / 10 for x in range(10)]
    df = pd.read_csv("t1_tpfp_scores.csv")
    class_threshold_df = pd.DataFrame()
    class_list = []
    thresh_list = []
    cc, num_files, class_files = load_voc_instances(dirname="/nfs/hpc/share/omorim/projects/PROB/data/VOC2007", split="train",
                                                    class_names=VOC_COCO_CLASS_NAMES["TOWOD"])
    for class_name in set(df.classes.values):
        index = VOC_COCO_CLASS_NAMES["TOWOD"].index(class_name)
        df_view = df.loc[df["classes"] == class_name]
        best_thresh = thresholds[1]
        best_score = 0
        best_prec = 0
        best_rec = 0
        for thresh in thresholds[1:]:
            prec, rec, score = fscore(thresh, df_view, cc[str(index)])
            #print(score)
            if score >= best_score:
                best_score = score
                best_prec = prec
                best_rec = rec
                best_thresh = thresh
        print("{} best thresh {}".format(class_name, best_thresh))
        print("num instances {}".format(cc[str(index)]))
        print("precision", best_prec, "recall", best_rec)
        class_list.append(class_name)
        thresh_list.append(best_thresh)
    class_threshold_df["class"] = class_list
    class_threshold_df["threshold"] = thresh_list
    print(class_threshold_df)
    class_threshold_df.to_csv("t1_known_class_f1_thresholds.csv")
    return 1


if __name__ == "__main__":
    main()
