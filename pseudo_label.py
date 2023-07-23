# Pseudolabelling file, save t1 model predictions on t2 for t2
# incremental learning

import xml.etree.ElementTree as ET
import pickle
from collections import defaultdict


t1_classes = set(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
              "train", "tvmonitor"])

VOC_CLASS_NAMES_COCOFIED = set([
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
])


def main():
    out_dr = "pseudolabels/t2/Annotations"
    image_box_hash = defaultdict(list)
    image_class_hash = defaultdict(list)
    for class_name in t1_classes:
        with open("pseudolabels/t2/known_50/boxes_{}.pickle".format(class_name), "rb") as fp:
            box_hash = pickle.load(fp)
            for k, v in box_hash.items():
                image_box_hash[k].append(v[0])
                image_class_hash[k].append(class_name)
    keys = set(image_box_hash.keys())
    # with open("pseudo_files_set.pickle", "wb") as fp:
    #     pickle.dump(keys, fp)
    # breakpoint()
    # return
    for count, image_id in enumerate(image_box_hash.keys()):
        if count % 1000 == 0:
            print(count)
        save_pseudo(image_id, image_box_hash[image_id], out_dr, image_class_hash[image_id])


def save_pseudo(image_id, pseudo_boxes, out_dr, class_names):
    """save the pseudo data to xml files in an Annotation folder
    takes in the output from class_agnostic pseudo labels"""
    # Add the object predictions to a new version of the xml file in out_dr
    # add pseudo labeled instances, and change t1 instances to unknown
    fn = "../PROB/data/VOC2007/Annotations/" + image_id + ".xml"
    tree = ET.parse(fn)
    root = tree.getroot()
    #print(fn, pseudo_boxes, class_names)
    #breakpoint()
    # Remove the annotations for previous classes by setting it as unknown
    for object in root.iter('name'):
        if object.text in t1_classes or object.text in VOC_CLASS_NAMES_COCOFIED:
            object.text = "unknown"
    for box, class_name in zip(pseudo_boxes, class_names):
        object_el = ET.SubElement(root, 'object')
        ET.SubElement(object_el, 'name').text = class_name
        bb_el = ET.SubElement(object_el, 'bndbox')
        ET.SubElement(bb_el, 'xmin').text = str(round(box[0]))
        ET.SubElement(bb_el, 'ymin').text = str(round(box[1]))
        ET.SubElement(bb_el, 'xmax').text = str(round(box[2]))
        ET.SubElement(bb_el, 'ymax').text = str(round(box[3]))
        ET.SubElement(object_el, 'difficult').text = "0"
    tree.write(out_dr + "/" + image_id + ".xml")


if __name__ == "__main__":
    main()
