python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_pretrained_known.py \
                          --num-gpus 1 \
                          --eval-only \
                          #train.init_checkpoint="./output/t1/known/pretrained_50_dino/model_0069999.pth"
                          train.init_checkpoint="./output/owdetr/t4/known/pretrained_50_dino_mixed_exemplar_full/model_0079999.pth"
