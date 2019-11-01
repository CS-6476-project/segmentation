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
  for index, img_path in enumerate(glob(os.path.join(test_img_dir, "*.jpg")), 1):
    print("\nImage %d" % index)
    img_base_name = os.path.basename(img_path)
    print("Running code for %s" % img_base_name)

    im = io.imread(img_path)
    hsv_im = rgb2hsv(im)

    # quantizeHSV(hsv_im, img_base_name)
    quantizeHSVAndPos(hsv_im, img_base_name)


def normalize_and_shuffle_data(data):
  data = normalizer.fit_transform(data)
  perm = np.arange(data.shape[0])
  np.random.shuffle(perm)
  data = data[perm]
  return data, np.argsort(perm)


def run_mean_shift(data, quantile=0.2):
  bandwidth = estimate_bandwidth(data, quantile, n_samples=100)
  ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data)

  return ms.labels_, ms.cluster_centers_.shape[0]


def save_as_mat_file(data, img_name, output_dir):
  obj_arr = np.empty((1,), dtype=np.object)
  obj_arr[0] = data
  savemat(os.path.join(output_dir, "%s.mat" % img_name), {segs_str: obj_arr})


def quantizeHSVAndPos(hsv_im, img_base_name):
  height, width, channels = hsv_im.shape
  # Technically should be height first followed by width, but doesn't matter
  x, y = np.meshgrid(np.arange(width), np.arange(height))
  hsv_im_with_pos = np.concatenate([hsv_im, x[..., None], y[..., None]], axis=2)

  hsv_pos_space_img = np.reshape(hsv_im_with_pos, (height * width, channels + 2))
  hsv_pos_space_img, inv_perm = normalize_and_shuffle_data(hsv_pos_space_img)

  labels, num_modes = run_mean_shift(hsv_pos_space_img, quantile=0.15)

  labels = labels[inv_perm]
  labels = np.reshape(labels, (height, width))
  hsv_pos_modes_dict[img_base_name] = num_modes
  print("Number of modes found by MeanShift for HSV + Pos space = %d\n" % num_modes)

  img_name = os.path.splitext(img_base_name)[0]
  save_as_mat_file(labels, img_name, hsv_pos_output_dir)


def quantizeHSV(hsv_im, img_base_name):
  height, width, channels = hsv_im.shape
  hsv_space_img = np.reshape(hsv_im, (height * width, channels))
  hsv_space_img, inv_perm = normalize_and_shuffle_data(hsv_space_img)

  labels, num_modes = run_mean_shift(hsv_space_img)

  labels = labels[inv_perm]
  labels = np.reshape(labels, (height, width))
  hsv_modes_dict[img_base_name] = num_modes
  print("Number of modes found by MeanShift for HSV space = %d\n" % num_modes)

  img_name = os.path.splitext(img_base_name)[0]
  save_as_mat_file(labels, img_name, hsv_output_dir)


if __name__ == '__main__':

  segs_str = 'segs'
  hsv_str = 'hsv'
  hsv_pos_str = 'hsv_pos'

  root_dir = '..'

  data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')

  output_dir = os.path.join(data_dir, segs_str, 'mean_shift')
  test_img_dir = os.path.join(data_dir, 'images', 'test')

  hsv_output_dir = os.path.join(output_dir, hsv_str)
  hsv_pos_output_dir = os.path.join(output_dir, hsv_pos_str)

  for directory in [hsv_output_dir, hsv_pos_output_dir]:
    if not os.path.exists(directory):
      os.makedirs(directory)

  hsv_modes_dict = {}
  hsv_pos_modes_dict = {}
  normalizer = StandardScaler()

  main()

  # with open(os.path.join(output_dir, '%s.pickle' % hsv_str), 'wb') as f:
  #   pickle.dump(hsv_modes_dict, f)
  #   print("\nPickle file created for HSV feature space")

  with open(os.path.join(output_dir, '%s.pickle' % hsv_pos_str), 'wb') as f:
    pickle.dump(hsv_pos_modes_dict, f)
    print("\nPickle file for HSV + Pos feature space")
