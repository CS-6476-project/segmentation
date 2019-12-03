 
import cv2
import os
import numpy as np
from glob import glob
from skimage import io
from skimage.color import rgb2hsv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
#  import matplotlib.pyplot as plt
import pickle

def main():
  # for index, img_path in enumerate(glob(os.path.join(test_img_dir, "*.jpg")), 1):
  #   print("\nImage %d" % index)
  #   print("Running code for %s" % img_path)
  #   quantizeRGB(img_path)
  quantizeRGB(os.path.join(test_img_dir, "207049.jpg"))


def normalize_and_shuffle_data(data):
  data = normalizer.fit_transform(data)
  perm = np.arange(data.shape[0])
  np.random.shuffle(perm)
  data = data[perm]
  return data, np.argsort(perm)
# #Loading original image
# originImg = cv2.imread('lenna.png')
# # Shape of original image    
# originShape = originImg.shape
# # Converting image into array of dimension [nb of pixels in originImage, 3]
# # based on r g b intensities    
# flatImg=np.reshape(originImg, [-1, 3])
# # Estimate bandwidth for meanshift algorithm    
# bandwidth = estimate_bandwidth(flatImg, quantile=0.2, n_samples=100)    
# ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
# # Performing meanshift on flatImg    
# ms.fit(flatImg)
# # (r,g,b) vectors corresponding to the different clusters after meanshift    
# labels=ms.labels_
# # Remaining colors after meanshift    
# cluster_centers = ms.cluster_centers_    
# # Finding and diplaying the number of clusters    
# labels_unique = np.unique(labels)    
# n_clusters_ = len(labels_unique)    
# print("number of estimated clusters : %d" % n_clusters_)    
# # Displaying segmented image    
# # segmentedImg = np.reshape(labels, originShape[:2])    
# # cv2.imshow('Image',segmentedImg)    
# # cv2.waitKey(0)    
# # cv2.destroyAllWindows()
# segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
# cv2.imshow('Image',segmentedImg.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def quantizeRGB(img_path):
  im = io.imread(img_path)
  rgb_im = im
  height, width, channels = rgb_im.shape
  rgb_space_img = np.reshape(rgb_im, (height * width, channels))
  rgb_space_img, inv_perm = normalize_and_shuffle_data(rgb_space_img)

  bandwidth = estimate_bandwidth(rgb_space_img, quantile=0.08, n_samples=100)
  ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(rgb_space_img)
  labels = ms.labels_
  labels = labels[inv_perm]
  labels = np.reshape(labels, (height, width))
  num_modes = ms.cluster_centers_.shape[0]
  print("Number of modes found by MeanShift = %d\n" % num_modes)

  obj_arr = np.empty((1,), dtype=np.object)
  obj_arr[0] = labels

  img_base_name = os.path.basename(img_path)
  rgb_modes_dict[img_base_name] = num_modes
  img_name = os.path.splitext(img_base_name)[0]
  savemat(os.path.join(output_dir, "%s.mat" % img_name), {segs_str: obj_arr})

if __name__ == '__main__':

  segs_str = 'segs'
  root_dir = '..'
  data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')
  test_img_dir = os.path.join(data_dir, 'images', 'test')
  output_dir = os.path.join(data_dir, segs_str, 'rgb-changes')

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  rgb_modes_dict = {}
  normalizer = StandardScaler()

  main()

  # with open(os.path.join(output_dir, 'rgb_modes_dict.pickle'), 'wb') as f:
  #   pickle.dump(rgb_modes_dict, f)
  #   print("\nPickle file rgb_modes_dict.pickle created")