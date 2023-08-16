import random

fn = "../PROB/data/VOC2007/ImageSets/Main/test.txt"

lines = []
with open(fn) as fp:
    lines = [line.rstrip() for line in fp]
sample = random.sample(lines, 100)

fn = "../PROB/data/VOC2007/ImageSets/Main/owod_test_sample.txt"
with open(fn, 'w') as f:
    for line in sample:
        f.write(f"{line}\n")
