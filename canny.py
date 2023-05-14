import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def check_canny():
    """Check canny edges were saved properly"""
    fn = "../PROB/data/VOC2007/ImageSets/Main/train.txt"
    dir = "../PROB/data/VOC2007/JPEGImages/"
    out_dir = "../PROB/data/VOC2007/CannyImages/"
    with open(fn) as fp:
        im_ids = fp.readlines()
    im_ids = [x.rstrip() for x in im_ids]
    for i, f in enumerate(im_ids):
        edge_file = out_dir + f + ".npy"
        file = dir + f + ".jpg"
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        with open(edge_file, "rb") as fp:
            edges_array = np.load(fp)
        edges = cv.Canny(img, 100, 200) / 255

        if not np.array_equal(edges, edges_array.astype("bool")):
            print(i, f)


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
            edges_array = edges / 255
            # plt.subplot(132),plt.imshow(edges,cmap = 'gray')
            # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            # jpg has compression, so does png...
            #cv.imwrite(out_dir+f+".png", edges)
            with open(out_dir+f+".npy", "wb") as fp:
                np.save(fp, edges_array.astype("bool"))
        #breakpoint()
        except:
            print("error")


if __name__ == "__main__":
    check_canny()
    #main()
