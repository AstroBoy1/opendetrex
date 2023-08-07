from detrex.config import get_config
from .models.dino_r50 import model

# get default config
dataloader = get_config("common/data/ood.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/t1/unknown/dino_pretrained_color_bilateral_edges/"

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
# Mixed precision training saves so much memory!
train.amp = dict(enabled=True)

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
model.select_box_nums_for_evaluation = 75

# Frequency channel
model.backbone.stem.in_channels = 4

# dropout parameters
model.transformer.encoder.attn_dropout = 0.1
model.transformer.encoder.ffn_dropout = 0.1
model.transformer.decoder.attn_dropout = 0.1
model.transformer.decoder.ffn_dropout = 0.1
model.criterion.matcher.cost_class = 1.0

model.dn_number = 1
model.num_queries = 75

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# Powers of 2
# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8

# dump the testing results into output_dir for visualization in tensorboard
dataloader.evaluator.output_dir = train.output_dir