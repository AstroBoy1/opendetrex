python tools/train_net.py --config-file projects/dino/configs/r50_4scale_unknown.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="./output/t1/unknown/bilateral_ft2/model_0039999.pth"