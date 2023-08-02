# Script to start a training experiment
#                           --dist-url tcp://127.0.0.1:12345

python tools/train_net.py --config-file projects/dino/configs/r50_4scale_unknown.py \
                          --num-gpus 2