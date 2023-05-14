import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def main():
    """Run canny edges on training images and use for the new loss function"""
    fn = "../PROB/data/VOC2007/ImageSets/Main/train.txt"
    dir = "../PROB/data/VOC2007/JPEGImages/"
    out_dir = "../PROB/data/VOC2007/CannyImages/"
    with open(fn) as fp:
        im_ids = fp.readlines()
    im_ids = [x.rstrip() for x in im_ids]
    for i, f in enumerate(im_ids):
        print(i, f)
        file = dir + f + ".jpg"
        try:
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            #assert img is not None, "file could not be read, check with os.path.exists()"
            # plt.subplot(131),plt.imshow(img,cmap = 'gray')
            # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            edges = cv.Canny(img, 100, 200)
            # plt.subplot(132),plt.imshow(edges,cmap = 'gray')
            # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            cv.imwrite(out_dir+f+".jpg", edges)
        #breakpoint()
        except:
            print("error")


if __name__ == "__main__":
    check_canny()
    #main()
