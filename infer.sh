python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_12ep.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="output/dino_agnostic_edges_pseudo_t1/model_0064999.pth"