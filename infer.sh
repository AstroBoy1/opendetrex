python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_12ep.py \
                          --num-gpus 2 \
                          --eval-only \
                          train.init_checkpoint="./dino_r50_4scale_12ep.pth"