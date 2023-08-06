python tools/train_net.py --config-file projects/dino/configs/r50_4scale_unknown.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="./output/t3/unknown/gpu_edges/model_0004999.pth"