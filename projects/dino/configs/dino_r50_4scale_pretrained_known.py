from detrex.config import get_config
from .models.dino_r50 import model
from pascal_voc_coco import register_pascal_voc
import itertools

# get default config
dataloader = get_config("common/data/ood.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep
train = get_config("common/train.py").train

# modify model config
# use the original implementation of dab-detr position embedding in 24 epochs training.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# modify training config
#train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
#train.output_dir = "./output/t1/known/pretrained_dino"

#train.init_checkpoint = "./output/t1/known/pretrained_50_dino/model_0069999.pth"
#train.output_dir = "./output/t2/known/pretrained_50_dino_mixed_exemplar_full"

#train.init_checkpoint="./output/t2/known/pretrained_50_dino/model_0059999.pth"
#train.init_checkpoint="./output/t2/known/pretrained_50_dino_pseudo/model_0079999.pth"
#train.output_dir = "./output/t1/known/pretrained_50_dino"
#train.output_dir = "./output/t2/known/pretrained_50_dino_pseudo"
#train.output_dir = "./output/t3/known/pretrained_50_dino_mixed_exemplar_full"
#train.output_dir = "./output/owdetr/t1/known/pretrained_dino_base"
#train.init_checkpoint = "./output/owdetr/t1/known/pretrained_dino/model_0199999.pth"
#train.init_checkpoint = "./output/owdetr/t2/known/pretrained_50_dino_mixed_exemplar/model_0044999.pth"
#train.output_dir = "./output/owdetr/t3/known/pretrained_50_dino_mixed_exemplar_full"
#train.output_dir = "./output/owdetr/t2/known/pretrained_50_dino_mixed_exemplar_full_44999"
#train.init_checkpoint = "./output/owdetr/t3/known/pretrained_50_dino_mixed_exemplar_full/model_0054999.pth"
#train.output_dir = "./output/owdetr/t4/known/pretrained_50_dino_mixed_exemplar_full"
#train.init_checkpoint = "./output/t3/known/pretrained_50_dino_mixed_exemplar_full/model_0054999.pth"
#train.init_checkpoint = "./output/t3/known/pretrained_50_dino_mixed_exemplar_full_55/model_0049999.pth"
#train.output_dir = "./output/t3/known/pretrained_50_dino_mixed_exemplar_full_55"
#train.init_checkpoint = "./output/t3/known/pretrained_50_dino_mixed_exemplar_full_55/model_0059999.pth"
#train.init_checkpoint = "./output/t4/known/pretrained_50_dino_mixed_exemplar_full/model_0029999.pth"
#train.output_dir = "./output/t4/known/pretrained_50_dino_mixed_exemplar_full_model_0029999"

#train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
#train.output_dir = "./output/d3/t2/pretrained_50_dino"
#train.output_dir = "./output/d3/t3/pretrained_50_dino"

#train.init_checkpoint = "./output/d3/t1/pretrained_50_dino/model_final.pth"
#train.output_dir = "./output/d3/t1/pretrained_50_dino_incremental_correct"

#train.init_checkpoint = "./output/d3/t1/pretrained_50_dino/model_final.pth"
#train.output_dir = "./output/d3/t1/pretrained_50_dino_incremental_nothing"

#train.init_checkpoint = "./output/d3/t1/pretrained_50_dino/model_final.pth"
#train.output_dir = "./output/d3/t1/pretrained_50_dino_exemplar"

#train.init_checkpoint = "./output/d3/t1/pretrained_50_dino/model_final.pth"
#train.output_dir = "./output/d3/t1/pretrained_50_dino_exemplar"

#train.init_checkpoint = "./output/d3/t2/pretrained_50_dino/model_final.pth"
#train.output_dir = "./output/d3/t2/pretrained_50_dino_exemplar"

#train.init_checkpoint = "./output/d3/t2/pretrained_50_dino/model_final.pth"
#train.output_dir = "./output/d3/t2/pretrained_50_dino_nothing"

#train.init_checkpoint = "./output/d3/t3/pretrained_50_dino/model_final.pth"
#train.output_dir = "./output/d3/t3/pretrained_50_dino_exemplar"

#train.init_checkpoint = "./output/d3/t3/pretrained_50_dino/model_final.pth"
#train.output_dir = "./output/d3/t3/pretrained_50_dino_incremental"

#train.init_checkpoint = "./output/d3/t3/pretrained_50_dino/model_final.pth"
#train.output_dir = "./output/d3/t3/pretrained_50_dino_nothing"

# max training iterations
#train.max_iter = 1900000
# 16551 / 4 = 4137.75 iterations per epoch
# 90000 / 4137.75 = 21 epochs
#train.max_iter = 90000
# 206888 iterations for 50 epochs
# 103444
train.max_iter = 206888
train.amp = dict(enabled=False)

# fast debug train.max_iter=20, train.eval_period=10, train.log_period=1
train.fast_dev_run.enabled = False

# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 100

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# 80 default classes for owod
model.num_classes = 80
model.select_box_nums_for_evaluation = 50
model.num_queries = 100

# modify optimizer config
optimizer.lr = 1e-5
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
#optimizer.params.lr_factor_func = lambda module_name: 1 if "backbone" in module_name else 1
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

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

OWDETR_T1_CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorbike","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
]

OWDETR_T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","diningtable",
    "pottedplant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","sofa"
]

OWDETR_T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake"
]

OWDETR_T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
]

towod_classes = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))
owdetr_classes = tuple(itertools.chain(OWDETR_T1_CLASS_NAMES, OWDETR_T2_CLASS_NAMES, OWDETR_T3_CLASS_NAMES, OWDETR_T4_CLASS_NAMES, UNK_CLASS))

dir = "../PROB/data/VOC2007"
register_pascal_voc("towod_t1", dir, "owod_t1_train", 2007, towod_classes, unknown=True, prev_known=0, exemplar=False, pseudo=False)
register_pascal_voc("towod_test_sample", dir, "owod_test_sample", 2007, towod_classes, unknown=True, prev_known=0, exemplar=False, pseudo=False)

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# Specify what directories to train and test on
#dataloader.train.dataset.names = "towod_t2"
dataloader.test.dataset.names = "towod_t1"
dataloader.train = dataloader.train_known

# modify dataloader config
dataloader.train.num_workers = 2

# Powers of 2
# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4

dataloader.evaluator.output_dir = train.output_dir
dataloader.evaluator.only_predict=False
dataloader.evaluator.previous_known=0
dataloader.evaluator.unknown=False
dataloader.evaluator.save_all_scores=False
dataloader.evaluator.upper_thresh=100
dataloader.evaluator.pseudo_label_known=False
dataloader.evaluator.single_branch=False
dataloader.evaluator.known_removal=False
#dataloader.predict_fn = "predictions/t1/known_dual_test.pickle"
#dataloader.tpfp_fn = "t2_known_tpfp_scores.csv"
#dataloader.unknown_predict_fn=""
dataloader.evaluator.num_classes = 20
#dataloader.pseudo_label_fn="pseudolabels/t2/known_50_2/"
dataloader.evaluator.all_classes = towod_classes