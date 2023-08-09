#                          train.init_checkpoint="./output/owdetr/t1/known/pretrained_dino/model_0199999.pth"

python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_pretrained_known.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="./output/owdetr/t2/known/pretrained_50_dino_mixed_exemplar/model_0049999.pth"
