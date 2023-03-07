#python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_12ep.py \
#                          --num-gpus 2 \
#                          --eval-only \
#                          train.init_checkpoint="./dino_r50_4scale_12ep.pth"
python tools/train_net.py \
  --config-file detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml \
  --eval-only MODEL.WEIGHTS ./model_final_b1acc2.pkl