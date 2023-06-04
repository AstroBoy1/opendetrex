# Pseudolabelling file

import pdb
import xml.etree.ElementTree as ET
import pickle
from collections import defaultdict
import numpy as np

# This file takes the nms predictions from open_eval.py pseudo branch and saves them to a file
# mixed with the ground truth


t1_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
              "train", "tvmonitor"]


def main():
    out_dr = "pseudolabels/t2/Annotations"
    image_box_hash = defaultdict(list)
    image_score_hash = defaultdict(list)
    image_class_hash = defaultdict(list)

    for class_name in t1_classes:
        print(class_name)
        with open("pseudolabels/t2/known_f1/boxes_{}.pickle".format(class_name), "rb") as fp:
            box_hash = pickle.load(fp)
            for k, v in box_hash.items():
                image_box_hash[k].append(v)
        with open("pseudolabels/t2/known_f1/scores_{}.pickle".format(class_name), "rb") as fp:
            score_hash = pickle.load(fp)
            for k, v in score_hash.items():
                image_score_hash[k].append(v)
                image_class_hash[k].append(class_name)
    for image_id in box_hash.keys():
        save_pseudo(image_id, image_box_hash[image_id], image_score_hash[image_id], out_dr, image_class_hash[image_id])


def save_pseudo(image_id, pseudo_boxes, pseudo_probs, out_dr, class_name):
    """save the pseudo data to xml files in an Annotation folder
    takes in the output from class_agnostic pseudo labels"""
    # Add the object predictions to a new version of the xml file in out_dr
    # add pseudo labeled instances, and remove t1 instances
    breakpoint()
    fn = "../PROB/data/VOC2007/Annotations/" + image_id + ".xml"
    tree = ET.parse(fn)
    root = tree.getroot()
    for box, prob in zip(pseudo_boxes, pseudo_probs):
        object_el = ET.SubElement(root, 'object')
        ET.SubElement(object_el, 'name').text = class_name
        bb_el = ET.SubElement(object_el, 'bndbox')
        ET.SubElement(bb_el, 'xmin').text = str(round(box[0]))
        ET.SubElement(bb_el, 'ymin').text = str(round(box[1]))
        ET.SubElement(bb_el, 'xmax').text = str(round(box[2]))
        ET.SubElement(bb_el, 'ymax').text = str(round(box[3]))
        ET.SubElement(object_el, 'difficult').text = "0"
    #tree.write(out_dr + "/" + image_id + ".xml")


if __name__ == "__main__":
    main()
