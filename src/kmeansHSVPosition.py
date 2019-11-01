import numpy as np
from sklearn.cluster import KMeans
import cv2
import pickle
import os
from scipy.io import savemat

def main():
    infile = open("hsv_position_modes_dict.pickle", "rb")
    kDict = pickle.load(infile)
    infile.close()


    for key in kDict:
        print(key)
        imgDir = os.path.join(test_img_dir, key)


        print(kDict[key])
        if kDict[key] < 2:
            kmeansHSVPosition(imgDir, 2)
        else:
            kmeansHSVPosition(imgDir, kDict[key])

def kmeansHSVPosition(imgDir, k):
    img = cv2.imread(imgDir)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dupImg = img[:]
    imgShape = dupImg.shape

    x, y = np.meshgrid(np.arange(0, np.size(hsvImage, 1)), np.arange(0, np.size(hsvImage, 0)))
    hsvImagex = np.concatenate([hsvImage, x[..., None]], axis=2)
    hsvImagey = np.concatenate([hsvImagex, y[..., None]], axis=2)
    pixels = np.reshape(hsvImagey, [-1, 5])

    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    centers = np.uint8(kmeans.cluster_centers_)
    labels = np.uint8(kmeans.labels_)
    mappedHues = centers[labels]
    resizedHues = mappedHues[:, 0:3]

    outputImg = resizedHues.reshape(img.shape)
    outputImg = cv2.cvtColor(outputImg, cv2.COLOR_HSV2RGB)

    img_base_name = os.path.basename(imgDir)
    img_name = os.path.splitext(img_base_name)[0]
    obj_arr = np.empty((1,), dtype=np.object)
    obj_arr[0] = outputImg
    savemat(os.path.join(output_dir, "%s.mat" % img_name), {'segs': obj_arr})
    # return [outputImg, centers]

if __name__ == '__main__':
    segs_str = 'KMeansHSVPositionSegs'
    root_dir = '..'
    data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')
    test_img_dir = os.path.join(data_dir, 'images', 'test')
    output_dir = os.path.join(root_dir, 'src', segs_str)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main()