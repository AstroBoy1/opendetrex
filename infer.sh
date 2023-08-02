python tools/train_net.py --config-file projects/dino/configs/r50_4scale_unknown.py \
                          --num-gpus 2 \
                          --eval-only \
                          train.init_checkpoint="./output/gpu_edges/model_0059999.pth"