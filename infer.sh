python tools/train_net.py --config-file projects/dino/configs/r50_4scale_unknown_adaptive_edges.py \
                          --num-gpus 2 \
                          --eval-only \
                          train.init_checkpoint="./output/owdetr/t1/unknown/scratch_adaptive_edges_color/model_0159999.pth"