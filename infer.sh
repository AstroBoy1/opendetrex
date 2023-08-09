python tools/train_net.py --config-file projects/dino/configs/r50_4scale_unknown.py \
                          --num-gpus 2 \
                          --eval-only \
                          train.init_checkpoint="./output/t1/unknown/scratch_adaptive_edges_color/model_final.pth"