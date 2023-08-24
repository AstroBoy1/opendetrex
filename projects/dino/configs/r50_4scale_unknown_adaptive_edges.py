from detrex.config import get_config
from .models.dino_r50 import model
from pascal_voc_coco import register_pascal_voc
import itertools

# get default config
dataloader = get_config("common/data/ood.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

#train.output_dir = "./output/t3/unknown/scratch_adaptive_edges_color_high_learning_rate"
#train.init_checkpoint = "./output/t3/unknown/scratch_adaptive_edges_color_high_learning_rate/model_0049999.pth"

# modify model config
# use the original implementation of dab-detr position embedding in 24 epochs training.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# max training iterations, batch size of 4, 16,551 examples
# t2 has 45,520 examples
# t3 has 39,402 examples
# t4 has 40,260 examples
# 16551 / 4 = 4137.75 iterations per epoch
# 90000 / 4137.75 = 21 epochs
#train.max_iter = 90000
# 206888 iterations for 50 epochs
# 103444
train.max_iter = 206888

# fast debug train.max_iter=20, train.eval_period=10, train.log_period=1
train.fast_dev_run.enabled = False
train.amp = dict(enabled=False)

# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 200

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# known and unknown class
model.num_classes = 2
model.select_box_nums_for_evaluation = 50

# Frequency channel
model.backbone.stem.in_channels = 4

# dropout parameters
#model.transformer.encoder.attn_dropout = 0.1
#model.transformer.encoder.ffn_dropout = 0.1
#model.transformer.decoder.attn_dropout = 0.1
#model.transformer.decoder.ffn_dropout = 0.1
model.criterion.matcher.cost_class = 1.0

model.dn_number = 1
model.num_queries = 100

# Adaptive edges
model.edges = True

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# Specify what directories to train and test on
#dataloader.train.dataset.names = "towod_t2"
dataloader.test.dataset.names = "towod_test_sample"
#dataloader.train = dataloader.train_known

# modify dataloader config
dataloader.train.num_workers = 2

# Powers of 2
# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4

# dump the testing results into output_dir for visualization in tensorboard
dataloader.evaluator.output_dir = train.output_dir

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

dataloader.evaluator.output_dir = train.output_dir
dataloader.evaluator.only_predict=False
dataloader.evaluator.previous_known=0
dataloader.evaluator.unknown=True
dataloader.evaluator.save_all_scores=False
dataloader.evaluator.upper_thresh=100
dataloader.evaluator.pseudo_label_known=False
dataloader.evaluator.single_branch=True
dataloader.evaluator.known_removal=False
#dataloader.unknown_predict_fn=""
dataloader.evaluator.num_classes = 20
dataloader.evaluator.all_classes = towod_classes
