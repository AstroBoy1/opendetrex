python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_t2.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="./output/t1/80_known/model_0064999.pth"
