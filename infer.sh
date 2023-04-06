python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_12ep.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="output/dino_general_dropout_augmentation/model_0024999.pth"