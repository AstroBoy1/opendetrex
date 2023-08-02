# Script to start a training experiment
#                           --dist-url tcp://127.0.0.1:12345

python tools/train_net.py --config-file projects/dino/configs/unknown_final.py \
                          --num-gpus 1