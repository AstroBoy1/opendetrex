echo "getting known predictions" &&
python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_pretrained_known.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="./output/t1/known/pretrained_50_dino/model_0069999.pth" \
                          dataloader.evaluator.save_all_scores=True \
                          dataloader.test.dataset.names="towod_test_sample" \
                          dataloader.evaluator.previous_known=0 \
                          dataloader.evaluator.tpfp_fn="pseudolabels/test_tpfp_scores.csv" &&
echo "generating tpfp scores" &&
python pseudolabel/t1_tpfp_scores.py pseudolabels/test_tpfp_scores.csv threshold_files/test_known50_class_f1_thresholds.csv owod_test_sample &&
echo "getting known prediction pseudo labels" &&
python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_pretrained_known.py \
                          --num-gpus 1 \
                          --eval-only \
                          train.init_checkpoint="./output/t1/known/pretrained_50_dino/model_0069999.pth" \
                          dataloader.test.dataset.names="towod_test_sample" \
                          dataloader.evaluator.previous_known=0 \
                          dataloader.evaluator.pseudo_label_known=True \
                          dataloader.evaluator.pseudo_label_fn="pseudolabels/test/" \
                          dataloader.evaluator.thresholds_fn="threshold_files/test_known50_class_f1_thresholds.csv"

echo "saving pseudo labels"
python pseudolabel/pseudo_label.py pseudolabels/test/Annotations pseudolabels/test/boxes_
class_names in t1_tpfp_scores.py should be adjusted accordingly

python tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_pretrained_known.py \
                          --num-gpus 1 \
                          train.init_checkpoint="./output/t1/known/pretrained_50_dino/model_0069999.pth" \
                          dataloader.evaluator.previous_known=0 \
                          dataloader.train.dataset.names = "towod_test_pseudo"
