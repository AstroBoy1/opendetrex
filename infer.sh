python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_single.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="./output/t1//single/model_0004999.pth"

                          