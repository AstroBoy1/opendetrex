from omegaconf import OmegaConf
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts
)
from open_eval.open_eval import PascalVOCDetectionEvaluator
from detrex.data import DetrDatasetMapper
from pascal_voc_coco import register_pascal_voc
import itertools

dataloader = OmegaConf.create()

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
ALL_CLASSES = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))
VOC_COCO_CLASS_NAMES["TOWOD"] = VOC_CLASS_NAMES

# 19 classes
OWDETR_T1_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bus", "car",
    "cat", "cow", "dog", "horse", "motorbike", "sheep", "train",
    "elephant", "bear", "zebra", "giraffe", "truck", "person"
]

# 21 classes
OWDETR_T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","diningtable",
    "pottedplant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","sofa"
]

# 20 classes
OWDETR_T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake"
]

# 20 classes
OWDETR_T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
]

VOC_COCO_CLASS_NAMES["OWDETR"] = tuple(itertools.chain(OWDETR_T1_CLASS_NAMES, OWDETR_T2_CLASS_NAMES, OWDETR_T3_CLASS_NAMES, OWDETR_T4_CLASS_NAMES, UNK_CLASS))

dir = "../PROB/data/VOC2007"
#dir = "pseudolabels/t2"

# Directories that specify the name of the files that contain which images
# to use for training and testing
register_pascal_voc("debug", dir, "debug", 2007, ALL_CLASSES)
register_pascal_voc("towod_t1", dir, "owod_t1_train", 2007, ALL_CLASSES)
register_pascal_voc("towod_t2", dir, "owod_t2_train", 2007, ALL_CLASSES)
register_pascal_voc("towod_t2_sample", dir, "owod_t2_train_sample", 2007, ALL_CLASSES)
register_pascal_voc("towod_t2_exemplar", dir, "owod_t2_train_andexemplars", 2007, ALL_CLASSES)
register_pascal_voc("towod_t3", dir, "owod_t3_train", 2007, ALL_CLASSES)
register_pascal_voc("towod_t3_exemplar", dir, "owod_t3_train_andexemplars", 2007, ALL_CLASSES)
register_pascal_voc("towod_t4", dir, "owod_t4_train", 2007, ALL_CLASSES)
register_pascal_voc("towod_test", dir, "test", 2007, ALL_CLASSES)

register_pascal_voc("towod_test_sample", dir, "owod_test_sample", 2007, ALL_CLASSES)

register_pascal_voc("owdetr_t1", dir, "owdetr_t1_train", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])
register_pascal_voc("owdetr_t1_sample", dir, "owdetr_t1_train_sample", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])
register_pascal_voc("owdetr_t2_exemplars", dir, "owdetr_t2_train_andexemplars", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])
register_pascal_voc("owdetr_t3_exemplars", dir, "owdetr_t3_train_andexemplars", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])
register_pascal_voc("owdetr_t2", dir, "owdetr_t2_train", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])
register_pascal_voc("owdetr_t3", dir, "owdetr_t3_train", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])
register_pascal_voc("owdetr_t4", dir, "owdetr_t4_train", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])
register_pascal_voc("owdetr_t4_exemplars", dir, "owdetr_t4_train_andexemplars", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])

register_pascal_voc("owdetr_test", "../PROB/data/VOC2007", "owdetr_test", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])
register_pascal_voc("owdetr_test_sample", "../PROB/data/VOC2007", "owdetr_test_sample", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])

#register_pascal_voc("owdetr_test", dir, "owdetr_test", 2007, VOC_COCO_CLASS_NAMES["OWDETR"])

register_pascal_voc("d3_test", dir, "d3_test", 2007, ALL_CLASSES)

pseudo_dir = "pseudolabels/d3/t2"
register_pascal_voc("d3_t1_incremental", "pseudolabels/d3/t1", "owod_t1_train", 2007, ALL_CLASSES)
register_pascal_voc("d3_t2_incremental", pseudo_dir, "owod_t1_train", 2007, ALL_CLASSES)
register_pascal_voc("d3_t3_incremental", "pseudolabels/d3/t3", "owod_t1_train", 2007, ALL_CLASSES)

# Augmentations to apply to the training data
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="towod_t1"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice"),
            L(T.RandomContrast)(
                intensity_min=0,
                intensity_max=2),
            L(T.RandomBrightness)(
                intensity_min=0,
                intensity_max=2),
            L(T.RandomSaturation)(
                intensity_min=0,
                intensity_max=2),
            L(T.RandomLighting)(
                scale=0.1
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice"),
            L(T.RandomContrast)(
                intensity_min=0, 
                intensity_max=2), 
            L(T.RandomBrightness)(
                intensity_min=0, 
                intensity_max=2), 
            L(T.RandomSaturation)(
                intensity_min=0, 
                intensity_max=2), 
            L(T.RandomLighting)(
                scale=0.1
            ),
        ],
        
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

# Augmentations to apply to the training data
dataloader.train_known = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="towod_t2"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice"),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice"),
        ],
        
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

# Augmentations to apply to the test data
dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="towod_test_sample", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

# Custom evaluation code for open world benchmark
dataloader.evaluator = L(PascalVOCDetectionEvaluator)(
    dataset_name="${..test.dataset.names}",
)
