 
import cv2
import os
import numpy as np
from glob import glob
from skimage import io
from skimage.color import rgb2hsv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
import pickle

def main():
  # for index, img_path in enumerate(glob(os.path.join(test_img_dir, "*.jpg")), 1):
  #   print("\nImage %d" % index)
  #   print("Running code for %s" % img_path)
  #   quantizeRGBPos(img_path)
  quantizeRGBPos((os.path.join(test_img_dir, "94095.jpg")))


def normalize_and_shuffle_data(data):
  data = normalizer.fit_transform(data)
  perm = np.arange(data.shape[0])
  np.random.shuffle(perm)
  data = data[perm]
  return data, np.argsort(perm)

def quantizeRGBPos(img_path):
  im = io.imread(img_path)
  x,y=np.meshgrid(np.arange(0,np.size(im,1)),np.arange(0,np.size(im,0)))
  originImgx = np.concatenate([im, x[...,None]], axis=2)
  rgb_im = np.concatenate([originImgx, y[...,None]], axis=2)
  height, width, channels = rgb_im.shape
  rgb_space_img = np.reshape(rgb_im, (height * width, channels))
  rgb_space_img, inv_perm = normalize_and_shuffle_data(rgb_space_img)

  bandwidth = estimate_bandwidth(rgb_space_img, quantile=0.05, n_samples=400)
  ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(rgb_space_img)
  labels = ms.labels_
  labels = labels[inv_perm]
  labels = np.reshape(labels, (height, width))
  num_modes = ms.cluster_centers_.shape[0]
  print("Number of modes found by MeanShift = %d\n" % num_modes)

  obj_arr = np.empty((1,), dtype=np.object)
  obj_arr[0] = labels

  img_base_name = os.path.basename(img_path)
  rgb_pos_modes_dict[img_base_name] = num_modes
  img_name = os.path.splitext(img_base_name)[0]
  savemat(os.path.join(output_dir, "%s.mat" % img_name), {segs_str: obj_arr})

if __name__ == '__main__':

  segs_str = 'segs'
  root_dir = '..'
  data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')
  test_img_dir = os.path.join(data_dir, 'images', 'test')
  output_dir = os.path.join(data_dir, segs_str, 'rgb_pos-changes')

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  rgb_pos_modes_dict = {}
  normalizer = StandardScaler()

  main()

  # with open(os.path.join(output_dir, 'rgb_pos_modes_dict.pickle'), 'wb') as f:
  #   pickle.dump(rgb_pos_modes_dict, f)
  #   print("\nPickle file rgb_pos_modes_dict.pickle created")