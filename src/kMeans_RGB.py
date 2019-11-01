import os
from glob import glob
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy.io import savemat
import pickle

def quantizeRGB(img_path, k):
    origImg = image.imread(img_path).astype(np.uint8);
    pixels = np.reshape(origImg, (-1,3)).astype('float');

    kMeans = KMeans(k).fit(pixels)
    clusterCenters = kMeans.cluster_centers_
    labels = kMeans.labels_
    labeledPixels = clusterCenters[labels]

    outputImg = np.reshape(labeledPixels, origImg.shape).astype('uint8');    

    mat_arr = np.empty((1,), dtype=np.object)
    mat_arr[0] = outputImg
    img_base_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_base_name)[0]
    savemat(os.path.join(output_dir, "%s.mat" % img_name), {'segs': mat_arr})

def quantizeRGBwithPos(img_path, k):
    origImg = image.imread(img_path).astype(np.uint8);
    dupImg = origImg[:]
    imgShape = dupImg.shape

    x,y = np.meshgrid(np.arange(0,np.size(origImg,1)),np.arange(0,np.size(origImg,0)))
    origImgx = np.concatenate([origImg, x[...,None]], axis=2)
    origImgy = np.concatenate([origImgx, y[...,None]], axis=2)
    pixels = np.reshape(origImgy, [-1, 5])

    kMeans = KMeans(k).fit(pixels)
    clusterCenters = kMeans.cluster_centers_
    labels = kMeans.labels_
    
    outputImg = np.reshape(labels, imgShape[:2]).astype('uint8')

    mat_arr = np.empty((1,), dtype=np.object)
    mat_arr[0] = outputImg
    img_base_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_base_name)[0]
    savemat(os.path.join(output_dir, "%s.mat" % img_name), {'segs': mat_arr})

    # segmentedImg = clusterCenters[outputImg]

def main():
    # infile = open("rgb_modes_dict.pickle", "rb")
    infile = open("rgb_pos_modes_dict.pickle", "rb")
    kDict = pickle.load(infile)
    infile.close()

    for key in kDict:
        print("\nImage %s" % key)
        img_path = os.path.join(test_img_dir, key)
        # quantizeRGB(img_path, kDict[key])
        quantizeRGBwithPos(img_path, kDict[key])

if __name__ == '__main__':

    root_dir = '..'
    data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')
    test_img_dir = os.path.join(data_dir, 'images', 'test')
    # output_dir = os.path.join(root_dir, 'src', 'k_means_RGB_files')
    output_dir = os.path.join(root_dir, 'src', 'k_means_RGB_pos_files')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main()


