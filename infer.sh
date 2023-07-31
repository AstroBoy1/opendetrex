python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_single.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="./output/t1/known/single/dino/model_0039999.pth"
