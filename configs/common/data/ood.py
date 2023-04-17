from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    DatasetMapper
)
from detectron2.evaluation import COCOEvaluator
from open_eval.open_eval import PascalVOCDetectionEvaluator
#from open_eval.open_world_eval import OWEvaluator
#from datasets.torchvision_datasets.open_world import OWDetection
from detrex.data import DetrDatasetMapper

from pascal_voc_coco import register_pascal_voc
import itertools

from torchvision.transforms import ColorJitter

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
# Used for the original dataset benchmark
ALL_CLASSES = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))
VOC_COCO_CLASS_NAMES["TOWOD"] = VOC_CLASS_NAMES

dir = "../PROB/data/VOC2007"
#dir = "../../datasets/VOCdevkit2007/VOC2007"
register_pascal_voc("towod_t1", dir, "train", 2007, VOC_COCO_CLASS_NAMES["TOWOD"])
register_pascal_voc("towod_test", dir, "test", 2007, ALL_CLASSES)


dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="towod_t1"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ), L(T.RandomContrast)(
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
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)
# dataloader.test = L(build_detection_test_loader)(
#     dataset=L(get_detection_dataset_dicts)(names="voc_2007_val", filter_empty=False),
#     mapper=L(DatasetMapper)(
#         augmentations=[
#             L(T.ResizeShortestEdge)(
#                 short_edge_length=800,
#                 max_size=1333,
#             ),
#         ],
#         is_train=False,
#         image_format="RGB",
#     ),
#     num_workers=4,
# )

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="towod_test", filter_empty=False),
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

# dataloader.test = L(build_detection_test_loader)(
#     dataset=L(get_detection_dataset_dicts)(names="voc_2007_test", filter_empty=False),
#     mapper=L(DetrDatasetMapper)(
#         augmentation=[
#             L(T.ResizeShortestEdge)(
#                 short_edge_length=800,
#                 max_size=1333,
#             ),
#         ],
#         augmentation_with_crop=None,
#         is_train=False,
#         mask_on=False,
#         img_format="RGB",
#     ),
#     num_workers=4,
# )


# class ARGS:
#     def __init__(self):
#         self.PREV_INTRODUCED_CLS = 0
#         self.val_root = "~/hpc-share/omorim/projects/PROB/data/OWOD"
#         self.test_set = "owod_all_task_test"
#         self.val_dataset = 'TOWOD'
#
#
# args = ARGS()
#
#
# dataset_val = OWDetection(args, args.val_root, image_set=args.test_set, dataset=args.val_dataset,
#                           transforms=None)
#
# dataloader.evaluator = L(OWEvaluator)(
#     voc_gt=dataset_val, iou_types=('bbox',)
# )

dataloader.evaluator = L(PascalVOCDetectionEvaluator)(
    dataset_name="${..test.dataset.names}",
)
# dataloader.evaluator = L(COCOEvaluator)(
#    dataset_name="${..test.dataset.names}",
# )
