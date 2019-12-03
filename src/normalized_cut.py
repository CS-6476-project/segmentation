import os
from glob import glob
from matplotlib import image
import numpy as np
from sklearn.cluster import KMeans
from skimage import data, segmentation, color
from skimage.future import graph
from skimage.color import rgb2hsv
from matplotlib import pyplot as plt
from scipy.io import savemat

def main():
  for index, img_path in enumerate(glob(os.path.join(test_img_dir, "*.jpg")), 1):
    print("\nImage %d" % index)
    img_base_name = os.path.basename(img_path)
    print("Running code for %s" % img_base_name)

    im = image.imread(img_path)
    # normalized_cut_RGB(im, img_base_name)

    hsv_im = np.uint8(rgb2hsv(im) * 255)
    normalized_cut_HSV(hsv_im, img_base_name)   
    

def save_as_mat_file(data, img_name, output_dir):
    obj_arr = np.empty((1,), dtype=np.object)
    obj_arr[0] = data
    savemat(os.path.join(output_dir, "%s.mat" % img_name), {segs_str: obj_arr})

def normalized_cut_RGB(im, img_base_name):
    labels1 = segmentation.slic(im, compactness=30, n_segments=400)
    # pixels = np.reshape(im, (-1,3)).astype('float')
    # kMeans = KMeans(400).fit(pixels)
    # clusterCenters = kMeans.cluster_centers_
    # labels1 = kMeans.labels_
    # out1 = np.reshape(labels1, im.shape[:2]).astype('uint8');
    out1 = color.label2rgb(labels1, img, kind='avg')

    g = graph.rag_mean_color(im, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
    ax[0].imshow(out1)
    ax[1].imshow(labels2)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    # img_name = os.path.splitext(img_base_name)[0]
    # save_as_mat_file(labels2, img_name, rgb_output_dir)

def normalized_cut_HSV(hsv_im, img_base_name):
    labels1 = segmentation.slic(hsv_im, compactness=30, n_segments=1000)
    # pixels = np.reshape(hsv_im, (-1,3)).astype('float')
    # kMeans = KMeans(400).fit(pixels)
    # clusterCenters = kMeans.cluster_centers_
    # labels1 = kMeans.labels_
    # out1 = np.reshape(labels1, hsv_im.shape[:2]).astype('uint8');
    out1 = color.label2rgb(labels1, hsv_im, kind='avg')

    g = graph.rag_mean_color(hsv_im, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)

    set_unique = set(list(np.reshape(labels2, -1)))
    print(len(set_unique))

    # img_name = os.path.splitext(img_base_name)[0]
    # save_as_mat_file(labels2, img_name, hsv_output_dir)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
    ax[0].imshow(out1)
    ax[1].imshow(labels2)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()


def normalized_cut_RGB_pos(im, img_base_name):
    dupImg = im[:]
    imgShape = dupImg.shape

    x,y = np.meshgrid(np.arange(0,np.size(im,1)),np.arange(0,np.size(im,0)))
    imX = np.concatenate([im, x[...,None]], axis=2)
    imY = np.concatenate([imX, y[...,None]], axis=2)
    pixels = np.reshape(imY, [-1, 5])

    kMeans = KMeans(k).fit(pixels)
    clusterCenters = kMeans.cluster_centers_
    labels = kMeans.labels_
    
    out1 = np.reshape(labels, imgShape[:2]).astype('uint8')

    g = graph.rag_mean_color(pixels, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g) 

    img_name = os.path.splitext(img_base_name)[0]
    save_as_mat_file(labels2, img_name, rgb_pos_output_dir)

if __name__ == '__main__':

  segs_str = 'segs'
  hsv_str = 'hsv'
  hsv_pos_str = 'hsv_pos'
  rgb_str = 'rgb'
  rgb_pos_str = 'rgb_pos'

  root_dir = '..'

  data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')
  segs_dir = os.path.join(data_dir, segs_str)

  nc_output_dir = os.path.join(segs_dir, 'normalized_cut')
  test_img_dir = os.path.join(data_dir, 'images', 'test')

  rgb_output_dir = os.path.join(nc_output_dir, rgb_str)
  rgb_pos_output_dir = os.path.join(nc_output_dir, rgb_pos_str)
  hsv_output_dir = os.path.join(nc_output_dir, hsv_str)
  hsv_pos_output_dir = os.path.join(nc_output_dir, hsv_pos_str)

  for directory in [rgb_output_dir, rgb_pos_output_dir, hsv_output_dir, hsv_pos_output_dir]:
    if not os.path.exists(directory):
      os.makedirs(directory)

#   hsv_modes_dict = {}
#   hsv_pos_modes_dict = {}
#   normalizer = StandardScaler()

  main()