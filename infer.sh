python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_pretrained_known.py \
                          --num-gpus 2 \
                          --eval-only \
                          train.init_checkpoint="./output/t2/known/pretrained_50_dino_pseudo/model_0079999.pth"

                          