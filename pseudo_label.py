# Pseudolabelling file

import pdb
import xml.etree.ElementTree as ET
import pickle


# This file takes the nms predictions from open_eval.py pseudo branch and saves them to a file
# mixed with the ground truth

def main():
    with open("pseudolabels/boxes.pickle", "rb") as fp:
        box_hash = pickle.load(fp)
    with open("pseudolabels/scores.pickle", "rb") as fp:
        probs_hash = pickle.load(fp)
    out_dr = "pseudolabels/t1/Annotations"
    for image_id in box_hash.keys():
        #breakpoint()
        # Just use the first 5 outputs
        save_pseudo(image_id, box_hash[image_id][:5], probs_hash[image_id][:5], out_dr, "pseudo_unknown")


def save_pseudo(image_id, pseudo_boxes, pseudo_probs, out_dr, class_name):
    """save the pseudo data to xml files in an Annotation folder
    takes in the output from class_agnostic pseudo labels"""
    # Add the object predictions to a new version of the xml file in out_dr
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
    tree.write(out_dr + "/" + image_id + ".xml")


if __name__ == "__main__":
    main()
