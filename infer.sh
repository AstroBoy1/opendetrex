python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_pretrained_known.py \
                          --num-gpus 2 \
                          --eval-only \
                          train.init_checkpoint="./output/d3/t2/pretrained_50_dino/model_final.pth"
