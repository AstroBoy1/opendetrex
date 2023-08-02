import random

with open("../PROB/data/VOC2007/ImageSets/Main/owod_t2_train.txt") as f:
    lines = f.read().splitlines()
    #breakpoint()
    new_list = random.sample(lines, k=5000)
    print(len(lines))
    with open('../PROB/data/VOC2007/ImageSets/Main/owod_t2_train_sample.txt', 'w') as fp:
        for line in new_list:
            fp.write(f"{line}\n")
