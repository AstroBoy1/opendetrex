python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_t2.py \
                          --num-gpus 2 \
                          --eval-only \
                          train.init_checkpoint="./output/t2/known_finetune_correct/model_0064999.pth"
