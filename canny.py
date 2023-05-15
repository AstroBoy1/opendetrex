import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time


def nearest_nonzero_idx(x, y, idx):
    # https://stackoverflow.com/questions/43306291/find-the-nearest-nonzero-element-and-corresponding-index-in-a-2d-numpy-array
    return idx[((idx - [x, y])**2).sum(1).argmin()]


def image_closest():
    """Check canny edges were saved properly"""
    fn = "../PROB/data/VOC2007/ImageSets/Main/train.txt"
    out_dir = "../PROB/data/VOC2007/CannyImages/"
    with open(fn) as fp:
        im_ids = fp.readlines()
    im_ids = [x.rstrip() for x in im_ids]
    for i, f in enumerate(im_ids):
        print(i, f)
        #breakpoint()
        edge_file = out_dir + f + ".npy"
        with open(edge_file, "rb") as fp:
            array = np.load(fp)
        one_indices = np.argwhere(array)
        array_shape = array.shape
        step_size = 10
        edge_grid = []
        for row in range(0, array_shape[0], step_size):
            row_vals = []
            for col in range(0, array_shape[1], step_size):
                cr, cc = (row + row + step_size) // 2, (col + col + step_size) // 2
                nearest_row, nearest_col = nearest_nonzero_idx(cr, cc, one_indices)
                row_vals.append((nearest_row, nearest_col))
            edge_grid.append(row_vals)

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
    # (8 rows, 9 columns)
    # array = np.array([[3, 2, 3, 3, 0, 2, 4, 2, 1],
    #    [0, 3, 4, 3, 4, 3, 3, 2, 0],
    #    [1, 3, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 1, 2, 0, 0, 2, 0, 0, 2],
    #    [3, 0, 0, 0, 0, 0, 0, 0, 1],
    #    [0, 0, 2, 2, 4, 4, 3, 4, 3],
    #    [2, 2, 2, 1, 0, 0, 1, 1, 1],
    #    [3, 4, 3, 1, 0, 4, 0, 4, 2]])
    array = np.ones((1000, 1000))
    row, col = 50, 379
    start = time.time()
    one_indices = np.argwhere(array)
    nearest_nonzero_idx(row, col, one_indices)
    end = time.time()
    print(end-start)
    # 0.06 seconds for a 1000x1000 image with all ones
    #image_closest()
