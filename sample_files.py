import random

fn = "../PROB/data/VOC2007/ImageSets/Main/owdetr_t1_train.txt"

lines = []
with open(fn) as fp:
    lines = [line.rstrip() for line in fp]
sample = random.sample(lines, 5000)

fn = "../PROB/data/VOC2007/ImageSets/Main/owdetr_t1_train_sample.txt"
with open(fn, 'w') as f:
    for line in sample:
        f.write(f"{line}\n")
