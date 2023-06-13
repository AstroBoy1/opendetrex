from detrex.config import get_config
from .models.dino_r50 import model
from detrex.config.configs.common.common_schedule import multistep_lr_scheduler

# get default config
dataloader = get_config("common/data/ood.py").dataloader
#dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
#lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
#lr_multiplier = multistep_lr_scheduler(values=[1.0, 0.1], warmup_steps=0, num_updates=90000)
train = get_config("common/train.py").train

# modify training config
#train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
#train.init_checkpoint = "./output/t1_known_90+29999/model_0029999.pth"
train.init_checkpoint = "./output/t1/known_scratch_ft3/model_final.pth"
train.output_dir = "./output/t1/known_scratch_ft4"

# use the original implementation of dab-detr position embedding in 24 epochs training.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

train.amp = dict(enabled=True)

# max training iterations
train.max_iter = 180000
#train.max_iter = 90000

# fast debug train.max_iter=20, train.eval_period=10, train.log_period=1
train.fast_dev_run.enabled = False

# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 20

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
model.num_classes = 20
#model.select_box_nums_for_evaluation = 100
model.num_queries = 100

# modify optimizer config
optimizer.lr = 1e-5
#optimizer.lr = 1e-5
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 8

# Powers of 2
# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8
#dataloader.train.total_batch_size = 4 

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
